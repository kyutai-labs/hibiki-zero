import React, { useState } from 'react';
import { View, Picker, StyleSheet } from 'react-native';

const LanguageSelector = () => {
    const [selectedLanguage, setSelectedLanguage] = useState('en');

    const handleLanguageChange = (itemValue: string) => {
        setSelectedLanguage(itemValue);
    };

    return (
        <View style={styles.container}>
            <Picker
                selectedValue={selectedLanguage}
                style={styles.picker}
                onValueChange={handleLanguageChange}
            >
                <Picker.Item label="English" value="en" />
                <Picker.Item label="Spanish" value="es" />
                <Picker.Item label="French" value="fr" />
                <Picker.Item label="German" value="de" />
                <Picker.Item label="Chinese" value="zh" />
            </Picker>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    picker: {
        height: 50,
        width: 150,
    },
});

export default LanguageSelector;
