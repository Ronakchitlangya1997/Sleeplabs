from django.urls import path, include
from SleepLabs.views import sleephighligh,Devicestatus,Sleeplabsgraph, home, deviceData, processSleepData, algov2, sleep_labs_graph_api_v3

urlpatterns = [
    path('', home, name="home"),
    path('deviceData/', deviceData, name="deviceData"),
    path('sleep_labs_graph_api_v3/', sleep_labs_graph_api_v3, name="sleep_labs_graph_api_v3"),
    path('processSleepData/', processSleepData, name="processSleepData"),
    path('algo/', algov2, name="algo"),
    path('Sleeplabsgraph/', Sleeplabsgraph, name="Sleeplabsgraph"),
    path('sleepAnalytics/', sleephighligh, name="sleephighligh"),
    path('devicestatus/', Devicestatus, name="Devicestatus"),
]
