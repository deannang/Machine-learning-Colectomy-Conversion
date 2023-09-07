library(readxl)
conversion <- read_excel("C:/Users/deann/PhD/Keegan/Conversion/Conversion dataset Sep 8 to send.xlsx")
tab<-table(conversion$T,conversion$Y)
prop.table(tab, 1)*100


apply(conversion[,2:4],2,function(x)t.test(x~conversion$Y))


tablefuncpercent<-function(x){
  table1<-table(conversion$Y,x)
  print(prop.table(table1,1)*100)
}

printchisq<-function(x){
  a<-chisq.test(conversion$Y,x)
  print(a)
}

apply(conversion[,2:31],2,function(x)printchisq(x))
apply(conversion[,2:31],2,function(x)tablefuncpercent(x))