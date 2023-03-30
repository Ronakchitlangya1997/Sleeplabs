from django.urls import path, include
from SleepLabs.views import SleeplabsAPI, sleep_labs_graph, sleep_labs_graph_api, home, deviceData,sleep_labs_graph_api_v2, processSleepData, algo,sleep_labs_graph_api_v3

urlpatterns = [
    path('sleep-labs', SleeplabsAPI, name="SleeplabsAPI"),
    path('', sleep_labs_graph, name="sleep_labs_graph"),
    path('sleep_labs_graph_api', sleep_labs_graph_api, name="sleep_labs_graph_api"),
    path('home', home, name="home"),
    path('deviceData/', deviceData, name="deviceData"),
    path('sleep_labs_graph_api_v2', sleep_labs_graph_api_v2, name="sleep_labs_graph_api_v2"),
    path('sleep_labs_graph_api_v3/', sleep_labs_graph_api_v3, name="sleep_labs_graph_api_v3"),
    path('processSleepData/', processSleepData, name="processSleepData"),
    path('algo/', algo, name="algo"),
]
