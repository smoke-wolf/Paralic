// Word list for eye-typing predictions, roughly in order of how common the
// words are in everyday English, plus words that matter in care situations.
// The first matches for the letters typed so far are offered as suggestions.

const COMMON = `
i you the to a and it is yes no please thank thanks help need want can what my me that
in of for be have not this are do on with we he she they your how now so but was at go
am will like just know get feel good okay ok hello hi bye water time here there when where
why who more some all one about up out if an or as by from them then than too very well
love would could should did does don't can't i'm it's that's let's think see look come make
take give tell talk say said call back home today tomorrow tonight morning night later soon
him his her hers our ours their theirs its myself yourself himself herself itself ourselves themselves
been being had has having were was doing done going gone went came coming got getting made making
know knew known thought seen saw looked looking wanted wanting needed told asked used using put
this these those which whose whom what whatever whenever wherever however whoever
into onto over under after before between through during without within along across against
around behind below above beside near far off down away again once only also still even ever
never always often sometimes usually really quite almost enough very much many most more less
little few several each every any some no none both either neither other another such same own
new old good bad great small big large long short high low right left next last first early late
young whole sure true real best better worse worst free full easy hard clear open close closed
people person man men woman women child children baby boy girl family friend friends mom mum dad
mother father parent parents wife husband son daughter brother sister grandma grandpa grandmother
grandfather aunt uncle cousin neighbour neighbor doctor nurse carer caregiver teacher
day days week weeks month months year years hour hours minute minutes second seconds moment
time times morning afternoon evening night tonight today tomorrow yesterday weekend holiday
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
spring summer autumn fall winter weather rain raining sun sunny snow wind windy cloudy cold hot warm cool
thing things something anything nothing everything someone anyone everyone nobody somebody
place places way ways world life house home room rooms kitchen bathroom bedroom garden door
window wall floor bed chair table desk sofa couch lamp light lights tv television radio music
phone computer tablet book books paper pen letter message email picture photo photos video film movie
car bus train taxi bike shop shopping store money work job school class lesson game games fun
food eat eating ate drink drinking drank water tea coffee juice milk soup bread sandwich breakfast
lunch dinner supper snack fruit apple banana orange chocolate cake ice cream sugar salt pepper meat
chicken fish rice pasta potato potatoes vegetables salad egg eggs cheese butter hungry thirsty full
tired sleepy sleep sleeping rest resting awake wake wakeup dream dreams nap
happy sad angry upset scared afraid worried nervous bored calm relaxed comfortable uncomfortable
excited proud lonely sorry glad fine great wonderful lovely beautiful funny silly strange
pain hurt hurts hurting ache aches sore sick ill unwell dizzy nauseous itch itchy hot cold fever
medicine medication pill pills tablet tablets appointment hospital clinic emergency ambulance
head face eyes eye ears ear nose mouth teeth tooth tongue lips neck shoulder shoulders arm arms
hand hands finger fingers chest back stomach tummy belly hip hips leg legs knee knees foot feet toe toes skin
breathe breathing breath cough sneeze swallow move moving turn turning lift sit sitting stand standing
lie lying walk walking run running fall fell push pull hold carry bring brought send sent
open close start stop wait finish finished begin began end try trying keep kept let
help helped helping ask answer question questions problem idea reason answer mistake
read reading write writing typing type spell listen listening hear heard watch watching
see seeing look looking show showed tell telling speak speaking talk talking say saying
remember forget forgot understand believe hope wish miss mean matter mind care decide choose
like liked love loved hate prefer enjoy want wanted need needed use used find found lose lost
buy bought sell pay paid cost spend spent save win won play played sing dance draw paint cook clean
wash washing shower bath toilet brush comb shave dress dressed clothes shirt trousers pants shoes
socks jacket coat hat glasses blanket pillow towel tissue tissues cup glass plate bowl spoon fork knife
straw bottle bag box key keys wallet charger battery remote
up down left right forward backward inside outside upstairs downstairs here there everywhere somewhere
yes no maybe please thanks thank okay sure alright fine sorry excuse pardon hello goodbye bye
morning goodnight welcome congratulations happy birthday merry christmas
number one two three four five six seven eight nine ten hundred thousand first second third half
news story stories song songs show shows program channel team match football soccer tennis
nature tree trees flower flowers grass bird birds dog dogs cat cats animal animals pet
city town country village street road park beach sea river lake mountain hill field sky star stars
planet planets moon earth space rocket
colour color red blue green yellow black white brown pink purple orange grey gray gold silver
important possible impossible different difficult simple special ready busy quiet loud slow fast
quick clean dirty wet dry heavy light dark bright soft hard warm cool safe dangerous
also because until while since unless although though whether
high higher highest hill hills hit history hire hide hidden hiking hip hips hint hint his him himself
absolutely actually anyway apparently basically certainly clearly definitely especially exactly finally generally
hopefully honestly instead mainly mostly naturally obviously perhaps possibly probably quickly rather recently
seriously simply slowly somehow suddenly totally truly unfortunately fortunately already ago anymore besides
otherwise therefore meanwhile together apart alone instead
i've i'd i'll you're you've you'd you'll he's she's we're we've we'll they're they've they'll isn't aren't
wasn't weren't hasn't haven't hadn't doesn't didn't won't wouldn't couldn't shouldn't there's here's what's
where's who's how's when's why's
accept add agree allow appear apply arrive attend avoid bake bear beat become became belong bend bite blow boil
borrow break broke broken build burn calm catch caught change changing charge check chew chose climb collect
comfort complain complete continue copy correct count cover cry crying cut deliver describe discuss disturb
drop dry earn enter escape expect explain feed feeding fill fit fix fixed follow fly forgive freeze gather grab
grow grew guess hang happen happened heal heat hide hit hug hurry imagine improve include invite join joke jump
kick kiss knock laugh laughing learn learning leave leaving lend lift live living lock marry measure meet meeting
mix nod notice offer order organise organize pack park pass pick plan plant point pour practise practice pray
prepare press pretend promise protect pull raise reach realise realize receive recognise recognize reduce refuse
relax remove repair repeat replace reply report return ride ring roll rub rush scratch scream search shake share
shout shut sign skip slip smell smile sneeze snore solve sort spill squeeze stay stick stir stretch study suggest
support suppose surprise swim switch taste teach test tickle tie touch travel treat trust undress unlock visit
vote wave wear weigh whisper wipe wonder worry wrap yawn yell
aching allergic allergy anxious anxiety bandage bleeding blood bruise burnt cramp cramps constipated diarrhea
diarrhoea diabetes exercise exercises faint fluid fluids gp infection inhaler injection insulin migraine mobility
oxygen painkiller painkillers paralysed paralyzed physio physiotherapy prescription pressure rash reposition
spasm spasms stiff suction swollen symptom symptoms temperature therapist therapy thermometer urine vomit vomiting
wheelchair walker hoist commode tube ventilator mask drip pump catheter dressing feeding numb tingling weak
alarm armchair bin blind blinds board bookshelf bucket button calendar candle carpet ceiling channel clock
cupboard curtain curtains cushion drawer duvet fan fridge freezer heater heating hoover iron kettle ladder laptop
mattress microwave mirror mug oven painting plug printer radiator rug scissors screen shelf sheet sheets sink soap
socket speaker stairs stove tap tape toaster toothbrush toothpaste torch toy toys tray umbrella vase wardrobe
bacon beans beef biscuit biscuits broccoli burger cabbage carrot carrots cereal cherries chips cookie cookies
corn crisps cucumber curry custard dessert donut grapes gravy ham honey jam jelly ketchup lemon lemonade lettuce
mango melon mushroom mushrooms noodles nuts oatmeal oats olive omelette onion onions pancake pancakes peach pear
peas pie pineapple pizza pork porridge pudding sausage sausages smoothie soda spaghetti steak strawberry
strawberries sweets toast tomato tomatoes tuna turkey vanilla wine beer yogurt yoghurt
adult adults boss colleague colleagues partner girlfriend boyfriend granddaughter grandson grandchildren nephew
niece stranger visitor visitors guest guests priest pastor chaplain volunteer
alone annoyed ashamed brave cheerful confident curious delighted depressed disappointed embarrassed exhausted
frustrated grateful grumpy guilty hopeful impatient jealous joyful miserable overwhelmed peaceful pleased positive
relieved restless safe satisfied shocked shy stressed surprised terrible thankful unhappy strong
airport bank cafe church city club gym hairdresser hotel library market mosque museum office pharmacy pool post
restaurant station supermarket synagogue temple theatre theater zoo
anniversary date easter festival future midday midnight noon past present schedule season weekday
cloud clouds forest leaf leaves ocean plant plants rock sunshine waves
art baking cards chess crossword dancing drawing gardening golf jigsaw knitting movies photography podcast puzzle
puzzles quiz rugby sewing singing sport sports swimming walking yoga
app apps calls camera chat contact contacts download emails headphones internet keyboard link login messages
mouse password post search settings text texts update upload videos volume wifi
amazing awful awesome boring brilliant broken careful cheap clever cosy cozy crazy cute dear delicious dirty empty
enormous expensive extra fair famous fantastic favourite favorite fresh friendly gentle gorgeous handsome healthy
heavy helpful honest horrible huge incredible interesting kind lazy loud lucky messy modern narrow nasty natural
nice noisy normal perfect pleasant polite poor popular pretty proper rare rich rough rude sharp shiny smooth sour
spicy sticky strict sweet tasty thick thin tidy tiny ugly unusual useful useless weird wide wild wise wrong
`;

export const WORDS = COMMON.split(/\s+/).map((w) => w.trim()).filter((w, i, all) => w && all.indexOf(w) === i);

// Likely first words when nothing has been typed yet.
export const STARTERS = ['I', 'Please', 'Can', 'Thank'];

// Next-word predictions: after one of these words, these are the likely follow-
// ups (ordered). Tuned for everyday and care conversation so the keyboard needs
// far fewer letters. Lowercase keys; matching is case-insensitive.
export const NEXT = {
  i: ['am', 'need', 'want', 'feel', 'would', 'can', "don't", 'like', 'will', 'have'],
  am: ['not', 'feeling', 'going', 'in', 'very', 'so', 'really', 'cold', 'hot', 'tired'],
  need: ['to', 'help', 'a', 'some', 'my', 'more', 'the'],
  want: ['to', 'a', 'some', 'my', 'the', 'you', 'more', 'that'],
  feel: ['good', 'better', 'bad', 'sick', 'tired', 'happy', 'sad', 'worried', 'cold', 'hot'],
  feeling: ['good', 'better', 'bad', 'sick', 'tired', 'worse', 'okay'],
  would: ['like', 'you'],
  like: ['to', 'a', 'some', 'you', 'the', 'this', 'that', 'more'],
  can: ['you', 'i', 'we', 'not', 'please'],
  could: ['you', 'i', 'we', 'please'],
  please: ['help', 'can', 'wait', 'come', 'bring', 'pass', 'turn', 'call', 'stop', 'open', 'close'],
  thank: ['you'],
  thanks: ['for', 'so'],
  the: ['bathroom', 'light', 'window', 'door', 'phone', 'water', 'blanket', 'pillow', 'doctor', 'nurse', 'tv', 'room'],
  to: ['the', 'go', 'eat', 'drink', 'sleep', 'rest', 'sit', 'lie', 'talk', 'see', 'bed', 'you'],
  my: ['family', 'medicine', 'glasses', 'phone', 'head', 'back', 'legs', 'arm', 'hand', 'pillow', 'mum', 'mom', 'dad'],
  a: ['little', 'lot', 'drink', 'break', 'minute', 'hug', 'question', 'doctor', 'nurse', 'tissue'],
  some: ['water', 'help', 'food', 'rest', 'quiet', 'company', 'more', 'tea', 'coffee'],
  is: ['it', 'there', 'that', 'this', 'too'],
  it: ['is', 'hurts', 'okay', 'again'],
  you: ['are', 'for', 'please', 'very', 'so', 'help'],
  are: ['you', 'we', 'they', 'okay'],
  we: ['can', 'should', 'could', 'go', 'are'],
  go: ['to', 'home', 'out', 'back', 'now', 'outside'],
  turn: ['on', 'off', 'up', 'down', 'it'],
  call: ['my', 'the', 'a', 'family', 'doctor', 'nurse'],
  bring: ['me', 'the', 'my', 'some'],
  pass: ['me', 'the', 'my'],
  open: ['the', 'it'],
  close: ['the', 'it'],
  this: ['is', 'one', 'hurts'],
  that: ['is', 'one', 'please', 'hurts'],
  not: ['now', 'yet', 'okay', 'good', 'sure'],
  "it's": ['too', 'okay', 'not', 'a'],
  "i'm": ['not', 'feeling', 'tired', 'okay', 'cold', 'hot', 'fine', 'sorry'],
  and: ['i', 'then', 'the', 'you', 'a', 'it', 'my', 'we', 'some', 'also'],
  but: ['i', 'it', 'not', 'the', 'you', 'now', 'please', 'thank'],
  or: ['not', 'a', 'the', 'maybe', 'later', 'something'],
  how: ['are', 'is', 'about', 'much', 'long', 'do', 'was', 'many', 'did'],
  what: ['is', 'time', 'are', 'do', 'did', 'about', 'happened', 'was', 'day'],
  where: ['is', 'are', 'did', 'do', 'am', 'can', 'were'],
  when: ['is', 'are', 'can', 'will', 'did', 'do', 'you', 'we'],
  why: ['is', 'are', 'did', 'not', 'do', 'can'],
  who: ['is', 'are', 'was', 'did', 'can', 'will'],
  have: ['a', 'you', 'to', 'some', 'been', 'the', 'we', 'my', 'any'],
  has: ['been', 'the', 'my', 'a', 'anyone', 'it'],
  had: ['a', 'my', 'some', 'enough', 'to', 'the'],
  do: ['you', 'not', 'it', 'we', 'that', 'this', 'i', 'something'],
  does: ['it', 'that', 'this', 'he', 'she', 'anyone'],
  did: ['you', 'it', 'i', 'we', 'he', 'she', 'they', 'not'],
  "don't": ['know', 'want', 'like', 'worry', 'feel', 'need', 'go', 'understand', 'think', 'forget'],
  "can't": ['breathe', 'see', 'hear', 'sleep', 'move', 'find', 'reach', 'wait', 'remember'],
  will: ['you', 'be', 'it', 'we', 'i', 'they', 'not'],
  "let's": ['go', 'talk', 'play', 'watch', 'eat', 'have', 'try', 'see'],
  let: ['me', 'us', 'it', 'them'],
  me: ['a', 'the', 'some', 'know', 'see', 'up', 'down', 'please', 'help', 'out'],
  get: ['me', 'the', 'up', 'some', 'a', 'my', 'better', 'help', 'out', 'ready'],
  give: ['me', 'it', 'the', 'my', 'some', 'up'],
  take: ['me', 'the', 'my', 'a', 'it', 'off', 'care'],
  put: ['the', 'it', 'my', 'on', 'me', 'down', 'away'],
  make: ['me', 'a', 'it', 'some', 'sure', 'the'],
  help: ['me', 'please', 'with', 'you'],
  tell: ['me', 'them', 'my', 'the', 'you', 'him', 'her'],
  show: ['me', 'you', 'the', 'my', 'them'],
  see: ['you', 'the', 'my', 'it', 'a', 'that', 'what'],
  look: ['at', 'for', 'up', 'out', 'after'],
  come: ['here', 'back', 'in', 'and', 'with', 'see', 'on', 'over'],
  going: ['to', 'home', 'out', 'well', 'on'],
  been: ['a', 'here', 'feeling', 'waiting', 'very', 'better', 'thinking'],
  be: ['a', 'there', 'careful', 'quiet', 'back', 'okay', 'better', 'here'],
  so: ['much', 'tired', 'good', 'happy', 'sorry', 'i', 'it', 'very', 'cold', 'hot'],
  very: ['much', 'good', 'tired', 'well', 'happy', 'cold', 'hot', 'sore', 'nice'],
  too: ['hot', 'cold', 'loud', 'much', 'bright', 'dark', 'fast', 'tight'],
  really: ['good', 'tired', 'sorry', 'want', 'need', 'like', 'hurts', 'happy'],
  of: ['the', 'my', 'water', 'tea', 'you', 'it', 'this', 'course'],
  in: ['the', 'my', 'a', 'pain', 'bed', 'here', 'there'],
  on: ['the', 'my', 'it', 'please', 'tv', 'a'],
  at: ['the', 'home', 'my', 'night', 'all', 'me'],
  for: ['me', 'you', 'the', 'a', 'my', 'helping', 'coming', 'everything'],
  with: ['me', 'you', 'my', 'the', 'a', 'it'],
  about: ['it', 'the', 'my', 'you', 'that', 'this', 'what'],
  from: ['the', 'my', 'here', 'you', 'there'],
  up: ['please', 'a', 'the', 'my', 'and'],
  down: ['please', 'a', 'the', 'my', 'here'],
  out: ['of', 'the', 'please', 'side', 'for'],
  off: ['the', 'please', 'it', 'my'],
  there: ['is', 'are', 'was', 'please'],
  here: ['is', 'please', 'are', 'now'],
  now: ['please', 'i', 'it', 'is'],
  today: ['i', 'is', 'was', 'please'],
  tomorrow: ['i', 'is', 'we', 'morning'],
  good: ['morning', 'night', 'afternoon', 'evening', 'idea', 'job', 'thank'],
  more: ['water', 'please', 'time', 'food', 'tea', 'of', 'than'],
  little: ['more', 'bit', 'less', 'water'],
  lot: ['of', 'better', 'more'],
  time: ['is', 'to', 'for', 'please'],
  "what's": ['the', 'that', 'wrong', 'happening', 'for', 'on'],
  "where's": ['my', 'the', 'everyone'],
  "how's": ['your', 'it', 'everyone'],
  love: ['you', 'it', 'that', 'this', 'to'],
  miss: ['you', 'my', 'home', 'them'],
  hurts: ['a', 'here', 'when', 'too', 'more'],
  pain: ['in', 'is', 'medicine', 'here'],
  hungry: ['please', 'now', 'can'],
  thirsty: ['please', 'can', 'now'],
  cold: ['please', 'can', 'here'],
  hot: ['please', 'can', 'here', 'water'],
  tired: ['now', 'please', 'today'],
  sorry: ['i', 'for', 'about'],
  excuse: ['me'],
  "i'll": ['be', 'try', 'wait', 'have', 'see'],
  "i've": ['been', 'got', 'had', 'finished'],
  "i'd": ['like', 'love', 'rather'],
  "you're": ['welcome', 'very', 'so', 'right', 'the'],
  "that's": ['good', 'great', 'right', 'fine', 'okay', 'not', 'enough', 'perfect'],
  watch: ['tv', 'a', 'the', 'something', 'it'],
  play: ['some', 'a', 'music', 'the', 'cards', 'games'],
  read: ['me', 'a', 'the', 'my', 'it'],
  listen: ['to', 'please'],
  talk: ['to', 'about', 'later', 'with'],
  sit: ['up', 'down', 'with', 'here'],
  lie: ['down'],
  move: ['me', 'my', 'the', 'it', 'over', 'up'],
  wait: ['a', 'for', 'please', 'here'],
  stop: ['please', 'it', 'that', 'the'],
  drink: ['of', 'please', 'water', 'some'],
  eat: ['something', 'now', 'please', 'later'],
  sleep: ['now', 'well', 'please', 'later'],
  your: ['help', 'name', 'day', 'family', 'phone'],
  our: ['family', 'house', 'home'],
  an: ['hour', 'ambulance', 'idea', 'apple', 'appointment'],
};
