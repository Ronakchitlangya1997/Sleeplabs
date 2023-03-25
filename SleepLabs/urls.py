from django.urls import path, include
from SleepLabs.views import SleeplabsAPI, sleep_labs_graph, sleep_labs_graph_api, home, deviceData

urlpatterns = [
    path('sleep-labs', SleeplabsAPI, name="SleeplabsAPI"),
    path('', sleep_labs_graph, name="sleep_labs_graph"),
    path('sleep_labs_graph_api', sleep_labs_graph_api, name="sleep_labs_graph_api"),
    path('home', home, name="home"),
    path('deviceData/', deviceData, name="deviceData"),
]
