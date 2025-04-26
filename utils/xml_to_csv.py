import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Dict
import os
import glob

def parse_sample_xml(xml_path: str) -> List[Dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    doc_id = root.attrib.get('id')
    source = root.attrib.get('source')
    url = root.attrib.get('url')
    
    # Focus and semantic tags
    focus = root.findtext('Focus')
    umls_cui = None
    umls_semantic_type = None
    umls_semantic_group = None
    umls = root.find('.//UMLS')
    if umls is not None:
        cui_elem = umls.find('.//CUI')
        if cui_elem is not None:
            umls_cui = cui_elem.text
        sem_type_elem = umls.find('.//SemanticType')
        if sem_type_elem is not None:
            umls_semantic_type = sem_type_elem.text
        sem_group_elem = umls.find('.//SemanticGroup')
        if sem_group_elem is not None:
            umls_semantic_group = sem_group_elem.text

    # Extract QAPairs
    qa_pairs = []
    for qapair in root.findall('.//QAPair'):
        qapair_pid = qapair.attrib.get('pid')
        question_elem = qapair.find('Question')
        answer_elem = qapair.find('Answer')
        if question_elem is not None and answer_elem is not None:
            question = question_elem.text.strip() if question_elem.text else ''
            question_qid = question_elem.attrib.get('qid')
            question_qtype = question_elem.attrib.get('qtype')
            answer = answer_elem.text.strip() if answer_elem.text else ''
            qa_pairs.append({
                'document_id': doc_id,
                'source': source,
                'url': url,
                'focus': focus,
                'umls_cui': umls_cui,
                'umls_semantic_type': umls_semantic_type,
                'umls_semantic_group': umls_semantic_group,
                'qapair_pid': qapair_pid,
                'question_qid': question_qid,
                'question_qtype': question_qtype,
                'question': question,
                'answer': answer
            })
    return qa_pairs

if __name__ == "__main__":
    # Recursively find all XML files in raw_data and subdirectories
    base_dir = "../MedQuAD/raw_data"
    xml_files = glob.glob(os.path.join(base_dir, '**', '*.xml'), recursive=True)
    print(f"Found {len(xml_files)} XML files to process.")

    all_qa_pairs = []
    errors = []
    for xml_path in xml_files:
        try:
            qa_pairs = parse_sample_xml(xml_path)
            all_qa_pairs.extend(qa_pairs)
            print(f"Processed {xml_path}: {len(qa_pairs)} Q&A pairs")
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            errors.append((xml_path, str(e)))

    df = pd.DataFrame(all_qa_pairs)
    out_path = "../MedQuAD/data/medquad_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"\nExtracted {len(df)} Q&A pairs from {len(xml_files)} files. Saved to {out_path}.")
    if errors:
        print(f"\nEncountered errors in {len(errors)} files:")
        for xml_path, err in errors:
            print(f"{xml_path}: {err}")
    else:
        print("No errors encountered.")
    print(df.head())
