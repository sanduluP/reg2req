import configparser


def create_config():
    config = configparser.ConfigParser()

    # Add sections and keπy-value pairs
    config['General'] = {
        'extraction': 'Sentences', # "Chunks" or "Sentences" 
        'bulk_load': False
    }

    config["LLM"] = {
        "requirement_extraction" : 'Mistral', # 'Mistral' or 'Internal_llama'
    }
    
    config["Graph"] = {
        "cypher_query" : """ 
            MATCH (n)-[r:is]->(req:Node{label:"requirement"}) 
            RETURN DISTINCT r.sentence  as sentence
        """
    }

    config['Data'] = {
        'data_source': 'SDS', # "DSA" or "SDS"
        'data_file': './data/SDS/20241015_MISSION_KI_Glossar_v1.0 en.pdf'
    }

    config['Retrieving'] = {
        'approach': 'Sparse Retrieval', # "Sparse Retrieval", "Dense Retrieval" or "Hybrid Retrieval"
        'k': 3
    }

    # Write the configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    create_config()