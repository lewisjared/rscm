from rscm.core import TimeseriesCollection


class TestTimeseriesCollection:
    def test_create(self, timeseries):
        collection = TimeseriesCollection()
        collection.add_timeseries("Test", timeseries)
        collection.add_timeseries("Other", timeseries)

        assert repr(collection) == '<TimeseriesCollection names=["Other", "Test"]>'

    def test_timeseries(self, timeseries):
        collection = TimeseriesCollection()
        collection.add_timeseries("Test", timeseries)

        ts = collection.timeseries()
        assert len(ts) == 1

        collection.add_timeseries("Other", timeseries)
        ts = collection.timeseries()
        assert len(ts) == 2

    def test_timeseries_immutable(self, timeseries):
        collection = TimeseriesCollection()
        collection.add_timeseries("Test", timeseries)

        # Modifications shouldn't be reflected in the initial object
        timeseries.set(0, 2.0)
        assert timeseries.at(0) == 2.0

        ts_from_collection = collection.get_timeseries_by_name("Test")
        assert ts_from_collection.at(0) == 1850.0
        ts_from_collection.set(0, 3.0)

        assert collection.get_timeseries_by_name("Test").at(0) == 1850.0
