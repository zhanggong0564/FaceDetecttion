//
// Created by 60180 on 2022/12/28.
//
#include <iostream>
using namespace std;

class Student{
public:
    explicit Student(int  age){
        m_age = age;
    }
    int& GetAge(){
        return m_age;
    }
    const int& GetAgeConst(){
        return m_age;
    }
    void showAge(){
        cout<<m_age<<endl;
    }
    void SetAge(int age){
        m_age = age;
    }
    //不能修改数据成员
    void SetAgeConst(int age) const{
        m_age = age;
    }
    //不能调用非const 成员函数
    void SetAgeConst2(int age) const{
        SetAge(age);
    }
    void SetAgeConst3(int age) const{
        SetAgeConst(age);
    }
private:
    int m_age=0;
};

int main(){
    Student stu(10);
    stu.GetAge() = 11;
    //stu.GetAgeConst() = 12; 编译报错
    stu.showAge();
    stu.SetAgeConst(10);
    return 0;
}