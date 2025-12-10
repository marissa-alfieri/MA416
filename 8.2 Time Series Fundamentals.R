# M08.2 - Time Series Fundamentals

mydata = read.csv("StockMarketB_9.25.csv")

plot(as.POSIXct(mydata$Time), mydata$Stock.Price, pch=16, col='blue')
lines(as.POSIXct(mydata$Time), mydata$Stock.Price, pch=19, col='black')

X = mydata$Time
length(X)

options(digits = 10)
Y <- mydata$Stock.Price
MA5 <- function(Y, t){
  mean(Y[(t-5):t])
}
Yma_7 <- MA5(Y, 7)
round(Yma_7, 4)
Yma_23 <- MA5(Y, 23)
round(Yma_23, 4)
Yma_79 <- MA5(Y, 79)
round(Yma_79, 4)

options(digits = 10)

Y <- mydata$Stock.Price

EMA_35 <- 0.32*Y[35] +
  0.2176*Y[34] +
  0.1480*Y[33] +
  0.1006*Y[32]

round(EMA_35, 4)






