from django.contrib import admin
from .models import PolicyDocument, ModerationResult, ViolationDetail

@admin.register(PolicyDocument)
class PolicyDocumentAdmin(admin.ModelAdmin):
    list_display = ['filename', 'user', 'file_size', 'uploaded_at']
    list_filter = ['uploaded_at', 'user']
    search_fields = ['filename', 'user__username']
    readonly_fields = ['uploaded_at']
    ordering = ['-uploaded_at']

class ViolationDetailInline(admin.TabularInline):
    model = ViolationDetail
    extra = 0
    readonly_fields = ['chunk_id', 'verdict', 'explanation', 'sources']
    can_delete = False

@admin.register(ModerationResult)
class ModerationResultAdmin(admin.ModelAdmin):
    list_display = [
        'filename', 'user', 'verdict', 'total_chunks',
        'violation_chunks', 'review_chunks', 'allowed_chunks', 'created_at'
    ]
    list_filter = ['verdict', 'created_at', 'user']
    search_fields = ['filename', 'user__username']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    inlines = [ViolationDetailInline]

@admin.register(ViolationDetail)
class ViolationDetailAdmin(admin.ModelAdmin):
    list_display = ['chunk_id', 'verdict', 'moderation_result']
    list_filter = ['verdict']
    search_fields = ['chunk_id', 'chunk_text', 'explanation']
    readonly_fields = ['moderation_result', 'chunk_id', 'chunk_text', 'verdict', 'explanation', 'sources']