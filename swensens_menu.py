# swensens_menu.py

swensens_menu = {
    "happy": ["Earthquake Sundae", "Mac 'n' Cheese", "BBQ Chicken Pizza", "Chicken Carbonara Pasta", "Crispy Calamari Rings", "Mango Madness Smoothie", "Mushroom Soup", "Grilled Fish Sambal"],
    "neutral": ["Caesar Salad", "Grilled Chicken Steak", "Fish & Chips", "Honey Butter Wings", "Clam Chowder Soup", "Classic Chicken Burger", "Mint Chocolate Chip Ice Cream", "Spaghetti Aglio Olio"],
    "sad": ["Mushroom Soup", "Baked Rice", "Classic Cheeseburger", "Tomato Basil Soup", "Beef Lasagna", "Chocolate Fudge Brownie", "Cottage Pie", "Banana Split"],
    "surprise": ["Chicken Satay Pizza", "Sirloin Steak", "Cheese Sticks", "Hot Pot Lamb Stew", "Durian Supreme Sundae", "Sweet Potato Fries", "Crispy Chicken Cutlet", "Spicy Seafood Marinara"],
    "anger": ["Chili Crab Pasta", "Spicy Buffalo Wings", "Firehouse Sundae", "Peri-Peri Grilled Chicken", "Sambal Prawns", "Cheesy Beef Nachos", "Sizzling Brownie with Ice Cream", "Red Velvet Cake"]
}

def recommend_menu(emotion, age, gender):
    """Recommend menu based on emotion, age, and gender."""
    menu_items = list(swensens_menu.get(emotion, ["Fries", "Ice Cream"]))

    # Add age-specific options
    if age < 12:
        # Kid-friendly items
        menu_items.extend(["Kids' Meal", "Chicken Nuggets", "Fruity Milkshake", "Mini Pancakes", "Cheesy Macaroni"])
    elif 12 <= age < 18:
        # Teenagers' favorites: energy-rich and popular choices
        menu_items.extend(["Spicy Chicken Wings", "Cheesy Fries", "Chocolate Milkshake", "Pepperoni Pizza", "Chicken Bolognese Pasta"])
    elif 18 <= age < 30:
        # Young Adults: trendy, balanced, and flavorful options
        menu_items.extend(["Loaded Nachos", "Classic Caesar Wrap", "Grilled Chicken Sandwich", "Iced Matcha Latte", "Double Mushroom Burger"])
    elif 30 <= age <= 50:
        # Adults: balanced meals, health-conscious choices
        menu_items.extend(["Grilled Salmon", "Quinoa Salad", "Vegetable Stir-Fry with Rice", "Stuffed Baked Potato", "Mixed Berry Smoothie"])
    elif age > 50:
        # Older Adults: lighter, nutritious, and easy-to-digest options
        menu_items.extend(["Healthy Salmon Salad", "Whole Wheat Pasta Primavera", "Fruit Parfait", "Grilled Veggie Plate", "Chicken Noodle Soup"])

    # Gender-specific preferences (do not accumulate)
    if gender == 'female':
        menu_items += ["Avocado Salad", "Spinach and Mushroom Frittata"]
    elif gender == 'male':
        menu_items += ["Ribeye Steak", "BBQ Beef Burger"]

    return list(set(menu_items))  # Remove duplicates