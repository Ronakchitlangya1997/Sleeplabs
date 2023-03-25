from django.db import models
import uuid
from Authentication.models import User

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
    timestamp = models.DateTimeField(auto_now_add=True)
    #Add the following at line 7
    DevID = models.CharField(max_length=20, default='None')
    class Meta:
        ordering = ('timestamp', )
