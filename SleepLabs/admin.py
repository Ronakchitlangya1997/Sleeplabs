from django.contrib import admin
from SleepLabs.models import SleepLab, SleepLabOptv1
# Register your models here

class SleepLabs_Admin(admin.ModelAdmin):

    list_display = ['id', 'AcX', 'AcY', 'AcZ', 'GyX', 'GyY', 'GyZ', 'OCC','DevID', 'timestamp']
admin.site.register(SleepLab, SleepLabs_Admin)

class SleepLabOptv1_Admin(admin.ModelAdmin):

    list_filter = ['DevID']
    list_display = ['auto_increment_id', 'timestamp', 'jsonData', 'DevID']
admin.site.register(SleepLabOptv1, SleepLabOptv1_Admin)