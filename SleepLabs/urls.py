from django.urls import path, include
from SleepLabs.views import SleeplabsAPI, sleep_labs_graph

urlpatterns = [
    path('sleep-labs', SleeplabsAPI, name="SleeplabsAPI"),
    path('', sleep_labs_graph, name="sleep_labs_graph")
]
