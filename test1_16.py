nums = [3,-10,-5,0,5,3,3,3,10,15,5,-20,-5,25,0]
a = len(nums) - 1
#print(a)
s = []
for i in range(0, a):
    for j in range(i + 1, a + 1):
        if nums[i] == nums[j]:
            s.append(i)
            break
b = len(s)
print(s)
for q in range(0,b):
    nums.pop(s[q]-q)
print(nums)