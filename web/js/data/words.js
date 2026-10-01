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
};
