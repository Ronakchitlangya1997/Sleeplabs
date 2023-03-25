from django.db import models

# Create your models here.
from django.contrib.auth.models import AbstractUser
# Create your models here.

class User(AbstractUser):
    Device_name = models.CharField(max_length=500, default='00000000')