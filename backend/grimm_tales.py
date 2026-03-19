"""Catalog of Grimm fairy tales (dark versions) for script generation."""

# Each tale: id, title, short description, approximate source text or synopsis
# These are the darker, original Grimm versions — NOT the sanitized children's versions.

TALES = [
    {
        "id": "the_juniper_tree",
        "title": "The Juniper Tree",
        "origin": "Brothers Grimm (1812)",
        "description": "A stepmother murders her stepson, chops him up, and serves him as stew to his father. The boy's spirit returns as a bird to exact revenge.",
        "themes": ["murder", "cannibalism", "revenge", "resurrection"],
        "synopsis": """A man's first wife dies after planting a juniper tree. His new wife despises her stepson and slams a heavy chest lid on his neck, decapitating him. She props his head back on, ties a kerchief around it, and places an apple in his hand so it looks like he is sitting normally. When her own daughter Marlene touches him and his head falls off, the stepmother convinces Marlene it was her fault. The stepmother then chops the boy into pieces, cooks him into a black pudding stew, and serves it to his father, who eats it unknowingly, declaring it delicious. Marlene gathers the bones and places them under the juniper tree. The boy's spirit rises as a beautiful bird that sings of the murder. The bird drops a millstone on the stepmother's head, killing her. The boy is restored to life.""",
    },
    {
        "id": "the_robber_bridegroom",
        "title": "The Robber Bridegroom",
        "origin": "Brothers Grimm (1812)",
        "description": "A young woman discovers her betrothed is a cannibal murderer who dismembers young women in his forest lair.",
        "themes": ["murder", "cannibalism", "deception", "justice"],
        "synopsis": """A miller promises his daughter to a wealthy suitor. The girl feels uneasy but visits his house deep in the forest as requested. A bird warns her to turn back. Inside, an old woman reveals that the bridegroom and his band murder and eat young women. The girl hides behind a barrel. The robbers drag in another maiden, force her to drink wine until her heart bursts, tear off her clothes, chop her body on a table, and salt the pieces. A severed finger with a gold ring flies through the air and lands in the hiding girl's lap. The old woman drugs the robbers with sleeping potion and helps the girl escape. At the wedding feast, the bride tells the story as if it were a dream, producing the finger with the ring as proof. The robber bridegroom turns pale. He and his band are executed.""",
    },
    {
        "id": "fitchers_bird",
        "title": "Fitcher's Bird",
        "origin": "Brothers Grimm (1812)",
        "description": "A sorcerer abducts women in a basket. Two sisters are dismembered for disobedience; the third outwits him and resurrects them.",
        "themes": ["abduction", "dismemberment", "cunning", "resurrection"],
        "synopsis": """A sorcerer disguised as a beggar touches young women and they must leap into his basket. He carries off three sisters, one by one. He gives each a key to a forbidden chamber and an egg to carry. The first two sisters open the chamber, find a basin full of dismembered bodies, drop the egg in blood (which won't wash off), and are themselves chopped to pieces and thrown in the basin. The third sister sets the egg aside first, opens the chamber, and reassembles her sisters' bodies, bringing them back to life. She hides them in a basket of gold, tricks the sorcerer into carrying them home, dresses a skull as a bride in the window, coats herself in honey and feathers to disguise herself as a strange bird, and escapes. Her brothers and kinsmen lock the sorcerer and his crew inside the house and burn it down.""",
    },
    {
        "id": "the_girl_without_hands",
        "title": "The Girl Without Hands",
        "origin": "Brothers Grimm (1812)",
        "description": "A father chops off his daughter's hands as part of a deal with the devil. She wanders the world, enduring suffering until divine grace restores her.",
        "themes": ["sacrifice", "suffering", "faith", "redemption"],
        "synopsis": """A poor miller unknowingly promises the devil whatever stands behind his mill — his daughter. When the devil comes to claim her, her tears keep her so clean and pure he cannot take her. He demands the father chop off her hands, and the terrified miller obeys. Still her tears purify the stumps, and the devil must give up. The handless girl wanders into a royal garden where an angel helps her eat pears from the king's trees. The king finds her, falls in love, and has silver hands made for her. They marry, but when the king goes to war, the devil intercepts their letters, forging a message that orders the queen and her baby killed. The queen mother hides them instead. The girl wanders for seven years in the wilderness, sheltered by angels, until her hands miraculously grow back. The king searches for years and finally finds his wife and child, whole and restored.""",
    },
    {
        "id": "bluebeard",
        "title": "Bluebeard",
        "origin": "Charles Perrault (1697) / Grimm variants",
        "description": "A wealthy lord with a blue beard gives his new wife a forbidden key. She discovers a chamber full of his murdered former wives.",
        "themes": ["curiosity", "murder", "forbidden knowledge", "rescue"],
        "synopsis": """A wealthy man with a hideous blue beard has had several wives, all of whom have vanished. He marries a young woman and gives her keys to every room in his castle, but forbids one small chamber. When he leaves on a journey, curiosity overwhelms her. She opens the forbidden room and finds the floor caked with blood and the bodies of his previous wives hung on the walls. She drops the key in the blood, and it will not wash clean. When Bluebeard returns and sees the stained key, he declares she must die like the others. She begs for time to say her prayers and sends her sister Anne to the tower to watch for their brothers. Just as Bluebeard raises his blade, the brothers arrive and cut him down. The wife inherits his fortune.""",
    },
    {
        "id": "the_singing_bone",
        "title": "The Singing Bone",
        "origin": "Brothers Grimm (1812)",
        "description": "A brother murders his sibling to claim credit for slaying a wild boar. Years later, a bone fashioned into a flute sings out the truth.",
        "themes": ["fratricide", "jealousy", "justice", "truth"],
        "synopsis": """A king offers his daughter's hand to whoever slays a monstrous boar ravaging the land. Two brothers set out — the younger, kind and humble; the elder, cunning and arrogant. A dwarf gives the younger brother a magic spear, and he kills the boar. On his way to the king, he meets his elder brother at an inn. The elder brother gets him drunk, murders him, buries the body under a bridge, and presents the boar to the king, claiming the prize and marrying the princess. Years later, a shepherd finds a bone under the bridge and carves it into a mouthpiece for his horn. When he blows it, the bone sings: 'Dear shepherd, you blow upon my bone; my brother slew me and buried me beneath the bridge-stone.' The king has the bridge dug up, finds the skeleton, and the elder brother confesses. He is sewn into a sack and drowned alive.""",
    },
    {
        "id": "the_death_of_the_little_hen",
        "title": "The Death of the Little Hen",
        "origin": "Brothers Grimm (1812)",
        "description": "A little hen chokes on a nut, and every creature who tries to help meets increasingly absurd and fatal misfortune in a chain of escalating death.",
        "themes": ["absurdity", "death", "chain reaction", "dark comedy"],
        "synopsis": """A little hen and a little cock go to the nut hill. The hen chokes on a nut kernel. The cock runs to the well for water; the well says he must first get red silk from the bride; the bride says she must first get a wreath from the willow; and so on in an escalating chain. By the time the cock returns with water, the hen is dead. The cock places her in a coffin and all the animals join the funeral procession. They must cross a stream; a straw lays itself across as a bridge but catches fire; a coal tries next but falls in and drowns; a stone lays itself across but slides into the water. One by one, every mourner falls in and drowns. The cock is left alone on the bank with the dead hen, digs her a grave, sits upon it, grieves himself to death, and then everyone is dead.""",
    },
    {
        "id": "the_willful_child",
        "title": "The Willful Child",
        "origin": "Brothers Grimm (1812)",
        "description": "The shortest and most disturbing Grimm tale: a disobedient child dies of illness, but keeps pushing its arm out of the grave until the mother beats it with a rod.",
        "themes": ["obedience", "death", "punishment", "horror"],
        "synopsis": """Once there was a child who was willful and did not do what its mother wished. For this reason, God had no pleasure in it and let it become ill, and no doctor could do it any good, and in a short time it lay on its deathbed. When it had been lowered into its grave and the earth spread over it, its little arm suddenly came out and reached upward. They pushed it back in and covered it with fresh earth, but that was no use — the little arm kept coming out again. The mother herself had to go to the grave and strike the little arm with a rod. When she had done that, the arm was drawn in, and at last the child had rest beneath the ground.""",
    },
    {
        "id": "the_six_swans",
        "title": "The Six Swans",
        "origin": "Brothers Grimm (1812)",
        "description": "A queen must sew six shirts from starflowers while keeping total silence for six years to save her brothers from an enchantment, even as she faces execution.",
        "themes": ["sacrifice", "silence", "perseverance", "false accusation"],
        "synopsis": """A king lost in the forest makes a deal with a witch: he will marry her daughter in exchange for guidance home. The witch's daughter, now queen, discovers the king has six sons from a previous marriage. The jealous step-queen sews enchanted shirts and throws them over the boys, turning them into swans. Their sister escapes and finds a fairy woman who tells her the only cure: she must sew six shirts from starflowers and not speak or laugh for six years. She begins her task in silence. A king from another land finds her in the forest and marries her despite her muteness. The king's mother steals the young queen's babies, smears blood on her mouth, and accuses her of eating them. Still the queen will not speak. She is sentenced to burn at the stake. On the day of execution, she has finished five shirts and nearly the sixth. As the fire is lit, six swans fly down. She throws the shirts over them and they transform back into her brothers — though the youngest still has a swan's wing because the last shirt lacked a sleeve. She finally speaks, reveals the truth, and the wicked mother-in-law is burned in her place.""",
    },
    {
        "id": "the_red_shoes",
        "title": "The Red Shoes",
        "origin": "Hans Christian Andersen (1845)",
        "description": "A vain girl becomes cursed to dance forever in her red shoes until she begs an executioner to chop off her feet.",
        "themes": ["vanity", "punishment", "repentance", "mutilation"],
        "synopsis": """Karen, a poor girl, is adopted by a rich old woman. She becomes vain and acquires a pair of beautiful red shoes. She wears them to church and to communion, thinking only of the shoes while she should be thinking of God. An old soldier with a red beard tells the shoes to stick fast when she dances. The shoes begin to dance of their own will and will not stop. Karen dances through fields and forests, through rain and shine, day and night. She cannot stop. She dances to the executioner's house and begs him to chop off her feet. He does, and the feet in the red shoes dance away into the forest. Karen gets wooden feet and crutches, and goes to work as a servant at the parsonage. She tries to go to church but the red shoes dance before the door, blocking her way. She weeps and repents deeply. An angel appears, and Karen's heart fills with so much sunshine and peace that it bursts, and her soul flies on sunshine to God, where no one asks about the red shoes.""",
    },
    {
        "id": "the_original_sleeping_beauty",
        "title": "Sun, Moon, and Talia (The Original Sleeping Beauty)",
        "origin": "Giambattista Basile (1634)",
        "description": "The original version of Sleeping Beauty, far darker than the fairy-tale version: the sleeping princess is assaulted by a king and gives birth while unconscious.",
        "themes": ["assault", "deception", "jealousy", "dark justice"],
        "synopsis": """A lord named Talia is told by wise men that his daughter will be endangered by a splinter of flax. He bans all flax from the castle, but one day the grown Talia sees an old woman spinning, touches the spindle, gets a splinter under her nail, and falls dead. Her father seats her on a velvet throne in a country estate and abandons the palace forever. A king out hunting finds the estate, discovers the sleeping Talia, and — unable to wake her — assaults her and leaves. Still asleep, Talia gives birth to twins: Sun and Moon. One baby sucks the flax splinter from her finger, and she wakes. The king returns, delighted, but he is already married. His jealous queen discovers the affair, orders the children kidnapped and cooked into a meal for the king, and tries to burn Talia alive. The cook secretly saves the children, substituting lamb. The king arrives just in time, learns the truth, has his queen burned in the fire meant for Talia, marries Talia, and rewards the cook.""",
    },
    {
        "id": "the_original_cinderella",
        "title": "Aschenputtel (The Original Cinderella)",
        "origin": "Brothers Grimm (1812)",
        "description": "The true Grimm Cinderella: stepsisters slice off parts of their feet to fit the slipper, and doves peck out their eyes at the wedding.",
        "themes": ["cruelty", "mutilation", "divine justice", "perseverance"],
        "synopsis": """Cinderella's dying mother tells her to be good and God will protect her. Her father remarries a cruel woman with two daughters who force Cinderella to work in the cinders. She plants a hazel twig on her mother's grave and waters it with tears until it grows into a tree where a white bird grants her wishes. When the king holds a three-day festival, Cinderella's stepmother dumps lentils in the ashes and says she may go only if she picks them all out; the birds help her. The tree gives her golden dresses and slippers. The prince dances only with her each night; she escapes twice. On the third night, the prince smears the stairs with pitch and catches her golden slipper. He goes house to house. The first stepsister cuts off her big toe to fit the shoe; the prince rides off with her but birds sing 'Look back! There's blood in the shoe!' He returns. The second stepsister slices off her heel. Again, the birds reveal the blood. Finally, Cinderella tries the slipper and it fits perfectly. At the wedding, doves swoop down and peck out both stepsisters' eyes, condemning them to blindness for their wickedness.""",
    },
]


def get_tale(tale_id: str) -> dict | None:
    for t in TALES:
        if t["id"] == tale_id:
            return t
    return None


def list_tales() -> list[dict]:
    return [{"id": t["id"], "title": t["title"], "origin": t["origin"],
             "description": t["description"], "themes": t["themes"]} for t in TALES]
