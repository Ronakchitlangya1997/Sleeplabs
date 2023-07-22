from django.urls import path, include
from .views import login, logout, getUserMobileApp

urlpatterns = [
    path('login', login, name="login"),
    path('logout', logout, name='logout'),
    path('getUserMobileApp/', getUserMobileApp, name='getUserMobileApp'),
]
