from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class student(BaseModel):
    name: str
    age: int = 18 # default value
    grade: Optional[str] = None # optional field
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=6, description="It represents the performance of the student in academics") 

new_student = student(name="John Doe", age="20", email="abc123@gmail.com", cgpa=5.6) # Pydantic will automatically convert the age from string to int and cgpa from string to float
print(new_student)

student_json = new_student.model_dump_json()
print(student_json)

student_dict = new_student.model_dump()
print(student_dict)