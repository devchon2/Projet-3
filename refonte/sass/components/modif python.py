import re

# Chemin vers le fichier à modifier
filename = '_buttons.scss'

# Définition des modifications à apporter
replacements = [
    (r'flex-direction:\s*row;', '@include row;'),
    (r'flex-direction:\s*column;', '@include column;\n  align-items: center;'),
    (r'font-size:\s*1.2rem;', 'font-size: 1.4rem;'),
    (r'font-size:\s*3rem;', 'font-size: 2.5rem;\n  margin-bottom: 20px;'),
    (r'font-size:\s*1rem;', 'font-size: 1.2rem;\n  margin-bottom: 20px;'),
    (r'width:\s*100%;', 'max-width: 300px;'),
    (r'margin:\s*0\s*auto;', 'margin: 0;'),
    (r'padding:\s*20px\s*0;', 'padding: 40px 20px;'),
    (r'max-width:\s*200px;', 'max-width: none;')
]

# Ouvrir le fichier et lire son contenu
with open(filename, 'r') as f:
    content = f.read()

# Appliquer les modifications
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Écrire le contenu modifié dans le fichier
with open(filename, 'w') as f:
    f.write(content)