# excess/enhancements/local sources
# rolling baseline for UUCON
# rolling baseline for mobile

# maybe rename to background?

def generate_baseline(data, window=dt.timedelta(hours=24), q=0.1):
    # data must have a datetime index
    baseline = (data.rolling(window=window, center=True).quantile(q)
                    .rolling(window=window, center=True).mean())

    return baseline