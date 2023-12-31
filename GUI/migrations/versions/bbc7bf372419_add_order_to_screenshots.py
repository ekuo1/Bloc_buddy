"""add order to screenshots

Revision ID: bbc7bf372419
Revises: 0861f0eb03e5
Create Date: 2023-05-03 20:52:38.264661

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bbc7bf372419'
down_revision = '0861f0eb03e5'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('screenshot', schema=None) as batch_op:
        batch_op.add_column(sa.Column('order', sa.Integer(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('screenshot', schema=None) as batch_op:
        batch_op.drop_column('order')

    # ### end Alembic commands ###
