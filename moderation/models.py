from django.db import models
from django.contrib.auth.models import User

class PolicyDocument(models.Model):
    """
    Stores uploaded policy documents
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='policy_documents')
    file = models.FileField(upload_to='policies/')
    filename = models.CharField(max_length=255)
    file_size = models.IntegerField(help_text="File size in bytes")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.filename} - {self.user.username}"
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Policy Document'
        verbose_name_plural = 'Policy Documents'

class ModerationResult(models.Model):
    """
    Stores moderation results for files checked against policies
    """
    VERDICT_CHOICES = [
        ('clean', 'Clean'),
        ('needs_review', 'Needs Review'),
        ('violation_found', 'Violation Found'),
        ('error', 'Error'),
    ]
    
    FINAL_VERDICT_CHOICES = [
        ('pending', 'Pending Review'),
        ('approved', 'Approved - Clean'),
        ('rejected', 'Rejected - Violation'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='moderation_results')
    file = models.FileField(upload_to='moderation_files/')
    filename = models.CharField(max_length=255)
    verdict = models.CharField(max_length=20, choices=VERDICT_CHOICES)
    final_verdict = models.CharField(
        max_length=20, 
        choices=FINAL_VERDICT_CHOICES, 
        default='pending',
        help_text="User's final decision after review"
    )
    total_chunks = models.IntegerField(default=0)
    allowed_chunks = models.IntegerField(default=0)
    review_chunks = models.IntegerField(default=0)
    violation_chunks = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.filename} - {self.verdict} - {self.user.username}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Moderation Result'
        verbose_name_plural = 'Moderation Results'

class ViolationDetail(models.Model):
    """
    Stores detailed violation information for each chunk
    """
    VERDICT_CHOICES = [
        ('violation', 'Violation'),
        ('review', 'Review'),
        ('unclear', 'Unclear'),
        ('error', 'Error'),
    ]
    
    moderation_result = models.ForeignKey(
        ModerationResult,
        on_delete=models.CASCADE,
        related_name='violations'
    )
    chunk_id = models.CharField(max_length=255)
    chunk_text = models.TextField()
    verdict = models.CharField(max_length=20, choices=VERDICT_CHOICES)
    explanation = models.TextField()
    sources = models.JSONField(default=list)
    
    def __str__(self):
        return f"{self.chunk_id} - {self.verdict}"
    
    class Meta:
        ordering = ['id']
        verbose_name = 'Violation Detail'
        verbose_name_plural = 'Violation Details'