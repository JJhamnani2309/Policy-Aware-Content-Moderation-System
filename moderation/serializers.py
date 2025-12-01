from rest_framework import serializers
from .models import PolicyDocument, ModerationResult, ViolationDetail

class PolicyDocumentSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = PolicyDocument
        fields = ['id', 'user', 'user_username', 'file', 'filename', 'file_size', 'uploaded_at']
        read_only_fields = ['id', 'user', 'uploaded_at']

class ViolationDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ViolationDetail
        fields = ['id', 'chunk_id', 'chunk_text', 'verdict', 'explanation', 'sources']

class ModerationResultSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source='user.username', read_only=True)
    violations = ViolationDetailSerializer(many=True, read_only=True)
    file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = ModerationResult
        fields = [
            'id', 'user', 'user_username', 'file', 'file_url', 'filename', 'verdict', 'final_verdict',
            'total_chunks', 'allowed_chunks', 'review_chunks', 'violation_chunks',
            'created_at', 'reviewed_at', 'violations'
        ]
        read_only_fields = ['id', 'user', 'created_at']
    
    def get_file_url(self, obj):
        request = self.context.get('request')
        if obj.file and hasattr(obj.file, 'url'):
            if request:
                return request.build_absolute_uri(obj.file.url)
            return obj.file.url
        return None

class ModerationResultListSerializer(serializers.ModelSerializer):
    """
    Simplified serializer for listing moderation results (without violations)
    """
    user_username = serializers.CharField(source='user.username', read_only=True)
    violation_count = serializers.IntegerField(source='violations.count', read_only=True)
    
    class Meta:
        model = ModerationResult
        fields = [
            'id', 'user_username', 'filename', 'verdict', 'final_verdict',
            'total_chunks', 'allowed_chunks', 'review_chunks', 'violation_chunks',
            'violation_count', 'created_at', 'reviewed_at'
        ]

class FinalVerdictSerializer(serializers.Serializer):
    """
    Serializer for updating final verdict
    """
    final_verdict = serializers.ChoiceField(choices=['approved', 'rejected'])