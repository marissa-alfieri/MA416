# M08.2 - Time Series Fundamentals

mydata = read.csv("StockMarketB_9.25.csv")

plot(as.POSIXct(mydata$Time), mydata$Stock.Price, pch=16, col='blue')
lines(as.POSIXct(mydata$Time), mydata$Stock.Price, pch=19, col='black')


