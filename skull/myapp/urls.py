from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.skull_home, name='skull_home'),
    path('detection', views.detection, name='detection'),
    path('reconstruction', views.reconstruction, name='reconstruction'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
    
    urlpatterns += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)