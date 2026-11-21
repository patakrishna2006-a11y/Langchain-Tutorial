from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

dotenv.load_dotenv()

llm1 = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

llm2 = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

prompt1 = PromptTemplate(
    template = "Generate a short and simple notes from the following text \n {text}",
    input_variables = ["text"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 short question from the following text \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | llm1 | parser,
    'quiz': prompt2 | llm2 | parser
})

merge_chain = prompt3 | llm1 | parser

chain = parallel_chain | merge_chain

text = """
Pokémon, short for Pocket Monsters, is a popular Japanese entertainment franchise created by Satoshi Tajiri. It revolves around fictional creatures called Pokémon, which humans can catch, train, and use in battles. The Pokémon world is filled with hundreds of unique species, each having different abilities, characteristics, and elemental types.

The story mainly follows Pokémon Trainers who travel across different regions, discover new Pokémon, challenge Gym Leaders, participate in tournaments, and build strong friendships with their Pokémon.

One of the most famous characters is Ash Ketchum, a young Pokémon Trainer from Pallet Town. Alongside his loyal partner Pikachu, Ash travels through various regions, including Kanto, Johto, Hoenn, Sinnoh, Unova, Kalos, Alola, and Galar. His ultimate dream is to become a Pokémon Master.

Pokémon come in different types, such as Fire, Water, Grass, Electric, Psychic, Dragon, and Flying. Each type has unique strengths and weaknesses. For example, Water-type Pokémon are generally strong against Fire-type Pokémon, while Electric-type Pokémon are effective against Water-type Pokémon.

Some popular Pokémon include Pikachu, Charizard, Bulbasaur, Squirtle, Eevee, Mewtwo, Lucario, and Greninja. Many Pokémon can evolve into stronger forms, gaining new abilities and sometimes changing their appearance or type.

Beyond battles and adventures, Pokémon teaches valuable lessons about friendship, teamwork, courage, determination, and never giving up. Its combination of exciting adventures, memorable characters, and imaginative creatures has made it one of the most successful entertainment franchises in the world.

"""

final_result = chain.invoke({"text":text})

print(final_result)
chain.get_graph().print_ascii()