def test_extract_street_name():
    assert extract_street_name("1 Ash Park Road SE54 3HB") == 'Ash Park Road'
    assert extract_street_name("Flat 14 1 Ash Park Road SE54 3HB") == 'Ash Park Road'
    assert extract_street_name("123 Main Blvd") == 'Main Blvd'
    assert extract_street_name("456 Maple AvEnUe") == 'Maple AvEnUe'
    assert extract_street_name("789 Oak Street") == 'Oak Street'

    # Additional test cases
    assert extract_street_name("42 Elm Drive") == 'Elm Drive'
    assert extract_street_name("15 Willow Ln") == 'Willow Ln'
    assert extract_street_name("789 Maple Terrace") == 'Maple Terrace'
    assert extract_street_name("10 Oak Cove") == 'Oak Cove'
    assert extract_street_name("675 Pine Circle") == 'Pine Circle'

    # Test with no street name in the address
    assert extract_street_name("Apartment 5, 27 Park Avenue") == ''

    # Test with only street number
    assert extract_street_name("1234") == ''

    # Test with empty address
    assert extract_street_name("") == ''