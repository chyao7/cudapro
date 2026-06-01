#include <iostream>

int main(){

    int a = 10;
    int b = a>>1;# 除以2向下取整
    std::cout << b << std::endl;
    for (int i=a/2; i > 0; i>>=1) {
        std::cout << i << std::endl;
    }
    return 0;
}