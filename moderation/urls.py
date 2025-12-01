from django.urls import path
from .views import (
    upload_policy_view,
    list_policies_view,
    delete_policy_view,
    clear_policies_view,
    moderate_file_view,
    moderation_history_view,
    moderation_detail_view,
    update_final_verdict_view
)

urlpatterns = [
    # Policy management
    path('upload-policy/', upload_policy_view, name='upload_policy'),
    path('policies/', list_policies_view, name='list_policies'),
    path('policies/<int:pk>/', delete_policy_view, name='delete_policy'),
    path('clear-policies/', clear_policies_view, name='clear_policies'),
    
    # Moderation
    path('moderate/', moderate_file_view, name='moderate_file'),
    path('history/', moderation_history_view, name='moderation_history'),
    path('history/<int:pk>/', moderation_detail_view, name='moderation_detail'),
    path('history/<int:pk>/verdict/', update_final_verdict_view, name='update_final_verdict'),
]