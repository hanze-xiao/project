def convert_to_fasta(protein_list):
    fasta_string = ""
    for i, protein in enumerate(protein_list):
        fasta_string += f">index_{i+1}\n"
        protein_sequence = "\n".join([protein[j:j+80] for j in range(0, len(protein), 80)])
        fasta_string += protein_sequence + "\n"
    return fasta_string

protein_list = ["ACDEFGHIJKLMNOPQRSTUVWXYZKSJHDKLJHFRKLSJDFLKNNSCUJKHUEHABUKBKNXCJKNAXKJHSDKHANSLKJDKLASJDBNJKASDHFBNUHBSNJKHDJKFHASJNNKNAKLJKALSHUJKFHAKJSHDASDASDASFASFDSAFD", "ACGTACGTACGTACGTACGTACGTACGT"]
fasta_content = convert_to_fasta(protein_list)

with open("output_davis.fasta", "w") as file:
    file.write(fasta_content)
