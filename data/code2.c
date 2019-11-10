int subtraction (int a, int b) {
   return a-b;
}
int main() {
   int (*fp) (int, int)=subtraction;
   int result = fp(5, 4);
   printf(" Using function pointer we get the result: %d",result);
   return 0;
}