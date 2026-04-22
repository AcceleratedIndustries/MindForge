class Mindforge < Formula
  include Language::Python::Virtualenv

  desc "Semantic memory engine for AI conversation transcripts"
  homepage "https://github.com/AcceleratedIndustries/MindForge"
  # Update on every release. The sdist filename follows mindforge_kb-X.Y.Z.tar.gz.
  url "https://files.pythonhosted.org/packages/source/m/mindforge-kb/mindforge_kb-0.2.0.tar.gz"
  sha256 "REPLACE_WITH_SDIST_SHA256"
  license "BUSL-1.1"

  depends_on "python@3.11"

  # Resources: regenerate with `poet -f mindforge-kb`. Keep exhaustive for
  # offline installs.
  resource "networkx" do
    url "https://files.pythonhosted.org/packages/source/n/networkx/networkx-3.3.tar.gz"
    sha256 "REPLACE_WITH_NETWORKX_SHA256"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/source/p/pyyaml/PyYAML-6.0.2.tar.gz"
    sha256 "REPLACE_WITH_PYYAML_SHA256"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "usage: mindforge", shell_output("#{bin}/mindforge --help")
  end
end
