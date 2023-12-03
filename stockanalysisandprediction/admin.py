from django.contrib import admin
from .models import Stock


@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ('name', 'date', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('name', 'date')
    search_fields = ('name',)