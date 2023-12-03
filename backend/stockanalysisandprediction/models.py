from django.db import models


class Stock(models.Model):
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.IntegerField()
    name = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.name} - {self.date}"
