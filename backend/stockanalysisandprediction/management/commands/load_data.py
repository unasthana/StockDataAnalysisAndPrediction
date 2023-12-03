import os
import pandas as pd
from django.core.management.base import BaseCommand
from stockanalysisandprediction.models import Stock


class Command(BaseCommand):
    help = 'Load a stocks csv file into the database'

    def handle(self, *args, **kwargs):
        # Stock.objects.all().delete() #DEBUG ONLY
        if not Stock.objects.exists():
            data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_stocks_5yr.csv'))
            data.dropna(subset=['open', 'high', 'low'], inplace=True)
            for _, row in data.iterrows():
                Stock.objects.create(
                    date=row['date'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    name=row['Name']
                )
            self.stdout.write(self.style.SUCCESS('Successfully loaded stock data'))
