import math
def abc(alist,blist,clist,dlist):
    D=[]
    E=[]
    global i,e,f
    i=0
    e=i+1
    f=e+1
    for i in range(len(alist)-2):
            for e in range(i+1,len(alist)-1):
                    for f in range(e+1,len(alist)):
                        if SimiTri(alist[i][0],alist[i][1],alist[e][0],alist[e][1],alist[f][0],alist[f][1],blist[i][0],blist[i][1],blist[e][0],blist[e][1],blist[f][0],blist[f][1])== True:
                            D.append(alist[i])
                            D.append(alist[e])
                            D.append(alist[f])
                            E.append(blist[i])
                            E.append(blist[e])
                            E.append(blist[f])
                            clist.append(D)
                            dlist.append(E)
                            break
                    if SimiTri(alist[i][0],alist[i][1],alist[e][0],alist[e][1],alist[f][0],alist[f][1],blist[i][0],blist[i][1],blist[e][0],blist[e][1],blist[f][0],blist[f][1])== True:
                        break
            if SimiTri(alist[i][0],alist[i][1],alist[e][0],alist[e][1],alist[f][0],alist[f][1],blist[i][0],blist[i][1],blist[e][0],blist[e][1],blist[f][0],blist[f][1])== True:
                break
    if SimiTri(alist[i][0],alist[i][1],alist[e][0],alist[e][1],alist[f][0],alist[f][1],blist[i][0],blist[i][1],blist[e][0],blist[e][1],blist[f][0],blist[f][1])== True:
        alist.remove(alist[i])
        alist.remove(alist[e-1])
        alist.remove(alist[f-2])
        blist.remove(blist[i])
        blist.remove(blist[e-1])
        blist.remove(blist[f-2])
    if len(alist)>=3 and len(D)!=0:
        abc(alist,blist,clist,dlist)
def abcd(alist,blist,clist,dlist):  #alist 是图一的单独点 bilst 是图一的组合
    global p
    D=[]
    E=[]
    for i in range(len(alist)):
        for e in range(len(blist)):
            if SimiTri(alist[i][0],alist[i][1],blist[e][0][0],blist[e][0][1],blist[e][1][0],blist[e][1][1],clist[i][0],clist[i][1],dlist[e][0][0],dlist[e][0][1],dlist[e][1][0],blist[e][1][1])== True:
                D.append(alist[i])
                E.append(clist[i])
                blist[e].extend(D)
                dlist[e].extend(E)
        
#def SimiTri([x0,y0],[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]):
def SimiTri(x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5):
  a1=math.sqrt((x2-x0)*(x2-x0)+(y2-y0)*(y2-y0))
  b1=math.sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0))
  c1=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
  a2=math.sqrt((x5-x3)*(x5-x3)+(y5-y3)*(y5-y3))
  b2=math.sqrt((x4-x3)*(x4-x3)+(y4-y3)*(y4-y3))
  c2=math.sqrt((x4-x5)*(x4-x5)+(y4-y5)*(y4-y5))

  cos0 = (a1*a1-b1*b1-c1*c1)/(-2*b1*c1)
  cos1 = (a2*a2-b2*b2-c2*c2)/(-2*b2*c2)
  Angel=math.degrees(math.acos((a1*a1-b1*b1-c1*c1)/(-2*b1*c1)))
  
  fabs1 = b1/c1 -b2/c2
  fabs1 = math.fabs(fabs1)
  fabs2 = cos0 - cos1
  fabs2 = math.fabs(fabs2)

  if fabs1< 0.1 and fabs2< 0.1:
    return True
  else:
    return False
def Merge(list0,list1,e0,e1):
  
    c0=[]                     #图一第一组三角形
    c0.append(list0[0])
    c1=[]#图一其他三角形
    d0=[]
    d0.append(list1[0])#图二第一组三角形
    d1=[]#图二其他三角形
    e00=[]#存放图一中和第一组三角形变化一致的
    e11=[]#存放图二中和第一组三角形变化一致的
    c00=list0[0]#用于三角形的合并
    d00=list1[0]#同上
    for i in range(1,len(list0)):
        c1 = list0[i]
        d1 = list1[i]
        if SimiTri(c0[0][0][0],c0[0][0][1],c0[0][1][0],c0[0][1][1],c1[0][0],c1[0][1],d0[0][0][0],d0[0][0][1],d0[0][1][0],d0[0][1][1],d1[0][0],d1[0][1]):
            c0.append(c1)
            d0.append(d1)
            c00=c00+c1
            d00=d00+d1
            #print(c0,1)
            #print(list0)
    for i in c0:
       list0.remove(i)
    for i in d0:
       list1.remove(i)
    e00.extend(c00)
    e11.extend(d00)     
    e0.append(e00)
    e1.append(e11)
    if len(list0)!=0:
        Merge(list0,list1,e0,e1)
    #return e0,e1
#print(m)
#print(abc(a)[1])

#for i in range(abc(a)[1]):
#    print(abc(a)[0])

            