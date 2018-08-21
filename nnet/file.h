#pragma once

template <class type_t>
class file
{
public:
	file();
	~file();

	static const std::vector<std::vector<type_t>> parseCSV(const std::string FN);
};

template<class type_t>
inline file<type_t>::file()
{
}

template<class type_t>
inline file<type_t>::~file()
{
}

template<class type_t>
const std::vector<std::vector<type_t>> file<type_t>::parseCSV(const std::string FILENAME)
{
	std::ifstream  data(FILENAME);
	std::string line;
	std::vector<std::vector<type_t>> parsedCsv;
	while (std::getline(data, line))
	{
		std::stringstream lineStream(line);
		std::string cell;
		type_t val;
		std::vector<type_t> parsedRow;
		while (std::getline(lineStream, cell, ','))
		{
			std::istringstream i(cell);
			if ((i >> val))
			{
				parsedRow.push_back(val);
			}
		}

		parsedCsv.push_back(parsedRow);
	}

	return parsedCsv;
}
