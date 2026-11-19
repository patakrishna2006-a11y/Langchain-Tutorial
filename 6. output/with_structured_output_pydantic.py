from langchain_ollama import ChatOllama
from typing import Optional
from pydantic import BaseModel, Field
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
)

class review(BaseModel):
    summary: str = Field(description="A concise summary of the review")
    sentiment: str = Field(description="The sentiment of the review can be positive, negative, or neutral")
    pros: Optional[list[str]] = Field(description="A list of the pros mentioned in the review")
    cons: Optional[list[str]] = Field(description="A list of the cons mentioned in the review")
    name: Optional[str] = Field(description="Only The name of the reviewer")

structured_llm = llm.with_structured_output(review)

result = structured_llm.invoke(
    """
    The Samsung Galaxy S24 stands out as a premium, highly portable flagship smartphone that delivers a stunning 6.2-inch, 120Hz Dynamic AMOLED display with a remarkably bright 2,600 nits of peak brightness. Performance is top-tier, driven by either a Qualcomm Snapdragon 8 Gen 3 or a Samsung Exynos 2400 processor depending on your region, which effortlessly handles multitasking and gaming while offering helpful Galaxy AI features and an industry-leading 7 years of software upgrades. The phone's battery life is solid, with its upgraded 4,000 mAh capacity easily lasting a full day of mixed use. Its main advantages lie in its pocket-friendly, durable IP68 design, excellent long-term support, and vibrant screen. However, it is held back by slow 25W charging speeds, a regional hardware split that can cause minor thermal throttling on Exynos models, the exclusion of a charging brick from the box, and the use of slower UFS 3.1 storage on the base 128GB model.
  Pros:
	• Industry-leading software support: Offers 7 years of Android OS and security upgrades, ensuring the phone stays relevant and secure well into the 2030s.
	• Excellent compact display: The 6.2-inch Dynamic AMOLED 2X screen is incredibly sharp, smooth with a 120Hz refresh rate, and exceptionally bright at 2,600 nits.
	• Premium, pocketable design: Features a highly durable, lightweight Armor Aluminum build with IP68 water resistance that is easy to use one-handed.
	• Useful Galaxy AI features: Packed with practical, built-in AI tools like Circle to Search, live call translation, and advanced photo editing.
	• Reliable battery life: The 4,000 mAh cell provides efficient energy management, comfortably lasting a full day of mixed use.
  Cons:
	• Slow charging speeds: Capped at just 25W wired charging, which lags far behind competitors that offer full charges in under 30 minutes.
	• Regional processor split: Depending on your country, you get either the Snapdragon 8 Gen 3 or the Exynos 2400, with the Exynos variant being slightly more prone to heating up during heavy gaming.
	• Downgraded base storage speed: The entry-level 128GB version uses older, slower UFS 3.1 storage instead of the faster UFS 4.0 found on the 256GB and 512GB models.
	• No charger in the box: Skimps on accessories, meaning you have to purchase a compatible fast-charging wall adapter separately.
	• Missing premium connectivity: Lacks Ultra-Wideband (UWB) support, making precise digital car key positioning and smart tag tracking less accurate compared to the Plus and Ultra models.

    Review by Krishna Pata
    
    """)

print(f"Summary: {result.summary} \n")
print(f"Sentiment: {result.sentiment} \n")
print(f"Pros: {result.pros} \n")
print(f"Cons: {result.cons} \n")
print(f"Name: {result.name} \n")