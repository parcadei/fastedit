const LIMIT = 10;


function Badge({ label }: { label: string }) {
    return <span className="badge">{label}</span>;
}

function Tag({ label }: { label: string }) {
    return <span className="tag">{label}</span>;
}


function Panel({ title }: { title: string }) {
    return (
        <div className="panel">
            <Badge label={title} />
        </div>
    );
}
