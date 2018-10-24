#include "MyForm.h"

using namespace std;
using namespace System;
using namespace System::Windows::Forms;

int main() {
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	trySudokuGA::MyForm form;
	Application::Run(%form);
};
