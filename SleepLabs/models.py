from django.db import models
import uuid
from Authentication.models import User

from django.db.models import JSONField

# Create your models here.
class SleepLab(models.Model):
    id = models.UUIDField(primary_key = True, default = uuid.uuid4, editable = False)
    AcX = models.CharField(max_length=100, default='None')
    AcY = models.CharField(max_length=100, default='None') 
    AcZ = models.CharField(max_length=100, default='None') 
    GyX = models.CharField(max_length=100, default='None') 
    GyY = models.CharField(max_length=100, default='None') 
    GyZ = models.CharField(max_length=100, default='None')
    OCC = models.CharField(max_length=100, default=1)
    timestamp = models.CharField(max_length=122, default='01 June, 2022')
    hrs = models.CharField(max_length=122, default='01 June, 2022')
    min = models.CharField(max_length=122, default='01 June, 2022')
    jsonData = JSONField(default=list)
    
    DevID = models.CharField(max_length=20, default='None')
    class Meta:
        ordering = ('-timestamp', )

class SleepLabOptv1(models.Model):
    auto_increment_id = models.AutoField(primary_key=True)
    timestamp = models.CharField(max_length=122, default='01 June, 2022')
    jsonData = JSONField(default=list)
    DevID = models.CharField(max_length=20, default='None')
    class Meta:
        ordering = ('-timestamp', )

class SleepLabOptv1Dummy(models.Model):
    auto_increment_id = models.AutoField(primary_key=True)
    timestamp = models.CharField(max_length=122, default='01 June, 2022')
    jsonData = JSONField(default=list)
    DevID = models.CharField(max_length=20, default='None')
    class Meta:
        ordering = ('-timestamp', )

