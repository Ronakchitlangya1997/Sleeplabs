from django.db import models
import uuid

# Create your models here.
class SleepLab(models.Model):
    id = models.UUIDField(primary_key = True, default = uuid.uuid4, editable = False)
    AcX = models.CharField(max_length=100, default='None')
    AcY = models.CharField(max_length=100, default='None') 
    AcZ = models.CharField(max_length=100, default='None') 
    GyX = models.CharField(max_length=100, default='None') 
    GyY = models.CharField(max_length=100, default='None') 
    GyZ = models.CharField(max_length=100, default='None')
    timestamp = models.DateTimeField(auto_now_add=True)
