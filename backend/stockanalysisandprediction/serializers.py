from rest_framework import serializers
from .models import Stock

class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stock
        fields = ['date', 'open', 'high', 'low', 'close', 'volume', 'name']