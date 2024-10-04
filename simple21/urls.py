"""
URL configuration for simple21 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from simple21.game.main import hello_world, print_instructions, set_user_name

urlpatterns = [
    path('admin/', admin.site.urls),
    path('test/', hello_world, name='hello_world'),
    path('instructions/', print_instructions, name='instructions'),
    path('set_user_name/', set_user_name, name='set_username')
]