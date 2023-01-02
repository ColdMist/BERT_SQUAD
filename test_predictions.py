from function_utilities import question_answer

context = """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, 
          songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing 
          and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. 
          Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. 
          Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, 
          earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy"."""


questions = ["For whom the passage is talking about?",
             "When did Beyonce born?",
             "Where did Beyonce born?",
             "What is Beyonce's nationality?",
             "Who was the Destiny's group manager?",
             "What name has the Beyoncé's debut album?",
             "How many Grammy Awards did Beyonce earn?",
             "When did the Beyoncé's debut album release?",
             "Who was the lead singer of R&B girl-group Destiny's Child?"]

answers = ["Beyonce Giselle Knowles - Carter", "September 4, 1981", "Houston, Texas",
           "American", "Mathew Knowles", "Dangerously in Love", "five", "2003",
           "Beyonce Giselle Knowles - Carter"]

for question, answer in zip(questions, answers):
  question_answer(context, question, answer)