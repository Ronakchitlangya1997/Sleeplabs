# Generated by Django 4.1.2 on 2022-12-04 05:49

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SleepLab',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('AcX', models.CharField(default='None', max_length=100)),
                ('AcY', models.CharField(default='None', max_length=100)),
                ('AcZ', models.CharField(default='None', max_length=100)),
                ('GyX', models.CharField(default='None', max_length=100)),
                ('GyY', models.CharField(default='None', max_length=100)),
                ('GyZ', models.CharField(default='None', max_length=100)),
                ('timestamp', models.DateField(auto_now_add=True)),
            ],
        ),
    ]
