from rest_framework import viewsets, generics
from .models import Stock
from .serializers import StockSerializer, StockNameSerializer


class StockViewSet(viewsets.ModelViewSet):
    queryset = Stock.objects.all()
    serializer_class = StockSerializer


class StockList(generics.ListAPIView):
    queryset = Stock.objects.values('name').distinct()
    serializer_class = StockNameSerializer


class StockDetail(generics.ListAPIView):
    serializer_class = StockSerializer

    def get_queryset(self):
        ticker = self.kwargs['ticker']
        return Stock.objects.filter(name=ticker)
