"""
Views for moderation API endpoints
"""
import os
import tempfile
from pathlib import Path
from rest_framework import status, generics
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from django.utils import timezone
from .models import PolicyDocument, ModerationResult, ViolationDetail
from .serializers import (
    PolicyDocumentSerializer,
    ModerationResultSerializer,
    ModerationResultListSerializer,
    FinalVerdictSerializer
)
from .modules.policy_store import (
    build_or_update_policy_store,
    load_policy_store,
    clear_policy_store,
    policy_store_exists
)
from .modules.moderation_engine import moderate_file_against_policy
import logging

logger = logging.getLogger('moderation')

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_policy_view(request):
    """
    Upload one or more policy PDFs and store them in the vector database.
    """
    try:
        files = request.FILES.getlist('files')
        
        if not files:
            return Response(
                {'error': 'No files provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"User {request.user.username} uploading {len(files)} policy files")
        
        # Save files to media directory and create PolicyDocument records
        saved_files = []
        file_paths = []
        
        for uploaded_file in files:
            # Save file
            policy_doc = PolicyDocument.objects.create(
                user=request.user,
                file=uploaded_file,
                filename=uploaded_file.name,
                file_size=uploaded_file.size
            )
            saved_files.append(policy_doc)
            file_paths.append(policy_doc.file.path)
            logger.info(f"Saved policy file: {uploaded_file.name}")
        
        # Build or update policy store
        try:
            policy_store = build_or_update_policy_store(file_paths)
            logger.info("Policy store updated successfully")
        except Exception as e:
            logger.exception("Error building policy store")
            # Delete the saved files if vector store creation fails
            for policy_doc in saved_files:
                policy_doc.delete()
            return Response(
                {'error': f'Error building policy store: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        serializer = PolicyDocumentSerializer(saved_files, many=True)
        
        return Response({
            'message': f'{len(files)} policy file(s) uploaded and processed successfully',
            'policy_documents': serializer.data,
            'policy_store_path': str(settings.POLICY_STORE_DIR)
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.exception("Error in upload_policy_view")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_policies_view(request):
    """
    List all policy documents for the current user.
    """
    try:
        policies = PolicyDocument.objects.filter(user=request.user)
        serializer = PolicyDocumentSerializer(policies, many=True)
        
        return Response({
            'count': policies.count(),
            'policies': serializer.data,
            'policy_store_exists': policy_store_exists()
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.exception("Error in list_policies_view")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_policy_view(request, pk):
    """
    Delete a specific policy document.
    Note: This does not remove the policy from the vector store.
    Use clear_policies_view to reset the entire policy store.
    """
    try:
        policy = PolicyDocument.objects.get(pk=pk, user=request.user)
        filename = policy.filename
        policy.delete()
        
        logger.info(f"User {request.user.username} deleted policy: {filename}")
        
        return Response({
            'message': f'Policy "{filename}" deleted successfully',
            'note': 'To fully reset the policy store, use the clear-policies endpoint'
        }, status=status.HTTP_200_OK)
        
    except PolicyDocument.DoesNotExist:
        return Response(
            {'error': 'Policy not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.exception("Error in delete_policy_view")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def clear_policies_view(request):
    """
    Clear all policy documents and reset the vector store.
    """
    try:
        # Delete all policy documents for the user
        user_policies = PolicyDocument.objects.filter(user=request.user)
        count = user_policies.count()
        user_policies.delete()
        
        # Clear the policy store (affects all users)
        clear_policy_store()
        
        logger.info(f"User {request.user.username} cleared {count} policies and reset policy store")
        
        return Response({
            'message': f'{count} policy document(s) deleted and policy store cleared',
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.exception("Error in clear_policies_view")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def moderate_file_view(request):
    """
    Moderate a single file against the stored policies.
    """
    try:
        # Check if file is provided
        if 'file' not in request.FILES:
            return Response(
                {'error': 'No file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES['file']
        logger.info(f"User {request.user.username} moderating file: {uploaded_file.name}")
        
        # Check if policy store exists
        if not policy_store_exists():
            return Response(
                {'error': 'Policy store is empty. Please upload policy documents first.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Load policy store
        try:
            policy_store = load_policy_store()
        except FileNotFoundError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        try:
            # Run moderation
            moderation_result_data = moderate_file_against_policy(
                policy_store,
                temp_file_path,
                uploaded_file.name,
                k=3
            )
            
            # Create ModerationResult record
            moderation_result = ModerationResult.objects.create(
                user=request.user,
                file=uploaded_file,
                filename=uploaded_file.name,
                verdict=moderation_result_data['verdict'],
                total_chunks=moderation_result_data['total_chunks'],
                allowed_chunks=moderation_result_data['allowed_chunks'],
                review_chunks=moderation_result_data['review_chunks'],
                violation_chunks=moderation_result_data['violation_chunks']
            )
            
            # Create ViolationDetail records
            for violation in moderation_result_data['violations']:
                ViolationDetail.objects.create(
                    moderation_result=moderation_result,
                    chunk_id=violation['chunk_id'],
                    chunk_text=violation['chunk_text'],
                    verdict=violation['verdict'],
                    explanation=violation['explanation'],
                    sources=violation['sources']
                )
            
            logger.info(f"Moderation complete for {uploaded_file.name}: "
                       f"verdict={moderation_result.verdict}")
            
            # Serialize and return result
            serializer = ModerationResultSerializer(moderation_result, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        logger.exception("Error in moderate_file_view")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def moderation_history_view(request):
    """
    Get moderation history for the current user.
    """
    try:
        results = ModerationResult.objects.filter(user=request.user)
        serializer = ModerationResultListSerializer(results, many=True)
        
        return Response({
            'count': results.count(),
            'results': serializer.data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.exception("Error in moderation_history_view")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def moderation_detail_view(request, pk):
    """
    Get detailed moderation result including all violations.
    """
    try:
        result = ModerationResult.objects.get(pk=pk, user=request.user)
        serializer = ModerationResultSerializer(result, context={'request': request})
        
        return Response(serializer.data, status=status.HTTP_200_OK)
        
    except ModerationResult.DoesNotExist:
        return Response(
            {'error': 'Moderation result not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.exception("Error in moderation_detail_view")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_final_verdict_view(request, pk):
    """
    Update the final verdict for a moderation result after user review.
    """
    try:
        result = ModerationResult.objects.get(pk=pk, user=request.user)
        
        serializer = FinalVerdictSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        final_verdict = serializer.validated_data['final_verdict']
        result.final_verdict = final_verdict
        result.reviewed_at = timezone.now()
        result.save()
        
        logger.info(f"User {request.user.username} set final verdict to {final_verdict} "
                   f"for moderation {pk}")
        
        return Response({
            'message': 'Final verdict updated successfully',
            'final_verdict': final_verdict
        }, status=status.HTTP_200_OK)
        
    except ModerationResult.DoesNotExist:
        return Response(
            {'error': 'Moderation result not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.exception("Error updating final verdict")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )