function Badge({ label }: { label: string }) {
    return <span className="badge">{label}</span>;
}


function Panel({ title }: { title: string }) {
    return (
        <div className="panel">
            <Badge label={title} />
        </div>
    );
}
