This project encompasses three main sections: economic forecasts, price action, and correlation.  There is web scraping, data cleaning, a variety of database formats, threading, multiprocessing, chat bots, visual charting, and a whole lot of data analytics. It was a long journey and I really learned a lot.


# data-forecasting

This section uses economic data forecasts to create buy/sell scores for currencies.  Forex Factory has a great calendar with data from all of the countries whose currencies make up the bulk of volume in the foreign exchange market. 

![pasted image 0](https://user-images.githubusercontent.com/62268115/121748495-b5606800-cace-11eb-8023-7b610c9f0078.png)        
Source: https://www.forexfactory.com/calendar

Accessing the calendar directly with Python is not possible so I used the Google Sheets API as an intermediary. The API made it possible to easily fill a database with historical data, with only minimal finnesse required to handle throttling and bad requests. 

Once the data was cleaned, the next step was to derive a relevance score for upcoming events in the calendar.  I wanted to calculate the importance that a data forecast may have to large funds and banks, thereby getting an idea for potential trades in the near future.  The premise for this is that (historically at least) interest rates drive currency flows, central banks set interest rates, and data drives central bank decision making.  Therefore, if I could determine the real importance of a given forecast I might gain a small directional edge.

Using normalized data, I applied the following calculations to each event:
- Find the difference between Forecast and Previous
- Apply an importance weighting (inverting some numbers, eg unemployment)
- Calculate the accuracy of the event’s Forecast in predicting the Actual
- Calculate the trend of the event’s recent Actuals
- Calculate the trend of the currency’s recent Actuals

After running each calendar event through these calculations I am left with a single rating for each forecast.   
![Screenshot from 2021-06-11 16-12-40](https://user-images.githubusercontent.com/62268115/121749256-f9a03800-cacf-11eb-9682-95c76eb4b5c6.png)

As useful as that may seem, I’ve been trading long enough to know that even the best ideas can end up being worthless.  In order to get an idea of the validity of my calculations, I needed to see if the ratings correlated to price movement. This can’t be done directly though, because with currencies you are always looking at the value of one against another. For example, EUR/CAD will show the value of the Euro against the Canadian dollar, but in order to see how the Euro’s forecast ratings affected it I needed to isolate EUR. The only way to do this is to make a currency index where the Euro is compared to everything else. 

In making the indexes I didn’t weight them according to trade volume, I simply normalized against volatility so that currencies with large daily ranges didn't overshadow those with small ranges. Below is an example of the Australian Dollar index with some hand-drawn SR lines.
![aud](https://user-images.githubusercontent.com/62268115/121749571-7df2bb00-cad0-11eb-90d7-f9a6d6a4a42c.png)

Now that I had the custom forecast data and a way to create currency indexes, I could plot the forecasts in the volume section (blue histogram) of the chart.
(ignore the green line here, thats from the correlation scanner)
![USD](https://user-images.githubusercontent.com/62268115/121750417-f60db080-cad1-11eb-9a34-fd8245500a1b.png)

That chart is showing 8hr candles. In reality these potential forecast trades should be looked at on no larger than a 1hr chart.  On those lower timeframe charts my custom formula showed potential but only on EUR, GBP and USD, the 3 currencies with the most economic data. 


# correlation-scanner

This section is used to find correlated markets for a given instrument. Ultimately all of the correlation data is combined to derive a single index line which will show if the underlying instrument might be under- or over-valued.  Since the process for scanning is essentially a nested loop with each instrument looking at every other instrument, I use threading to read all data into memory up front.  Once that's done, multiprocessing is used to handle the top level loop and threading for the nested loops.  So each instrument is getting its own process, and inside that process is a thread executor to scan all the other instruments.  

The first logic hurdle to overcome was differing market hours.  My focus was on currencies which are open 24/5, however many of the instruments I use for correlation scanning are only open during the NY session, or about 8/5.  This would have caused much lower correlation scores for those markets and I didn't want that. I decided to align the data on their datetime indexes and only scan periods where both markets were open. 

Once the base instrument (the one that is scanning for correlated markets) is aligned with its corr instrument (the one being scanned), various shift periods are scanned. This is done to find the markets which a regular correlation function might miss.  For example, if Oil tends to lead the price of CAD by n periods, a straightforward scan might return a low correlation score.  However, shifting Oil back by n periods could result in a much higher score.  Also, some of the corr instruments are actually custom spreads.

After the best shift value is found, only the rows (periods) where correlation breaches the minimum threshold are saved. This results in a somewhat "gappy" Series filled with nans, however this is fine since it will get averaged together with many other corr instruments.  

Once all the corr markets have been scanned, the different Series are combined into a Dataframe. From there, I take the average of each to end up with a Dataframe consisting of 2 columns: normalized Close of the base instrument, and the normalized Close of the corr markets. The latter gives me a synthetic index which I can use to evaluate the base instrument.  I don't technically need to save the normalized Close of the base instrument, but I do this so I can easily check the final correlation score of the sythentic index.

The image below shows a synthetic index overlayed on a currency chart.  The strategy is to trade divergence between the candles and the line, with the assumption that the line is right and the candles are wrong.  That doesn't always end up being the case, but the reasoning is that if all other markets which are typically correlated to AUDJPY are rising while AUDJPY is falling, its more likely that a single market is wrong than all the others.

![Screenshot from 2021-06-11 16-37-55](https://user-images.githubusercontent.com/62268115/121751350-8d273800-cad3-11eb-9c96-67d79f85c121.png)


