from django.contrib import admin
from SleepLabs.models import SleepLab
# Register your models here

class SleepLabs_Admin(admin.ModelAdmin):

    list_display = ['id', 'AcX', 'AcY', 'AcZ', 'GyX', 'GyY', 'GyZ', 'OCC', 'timestamp']
admin.site.register(SleepLab, SleepLabs_Admin)