#pragma once

class file
{
public:
	file();
	~file();

	static const std::vector<std::vector<double>> parseCSV2(const std::string FILENAME);
};
